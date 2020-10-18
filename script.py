import argparse
from models.random_model_trainer import RandomModelTrainer

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--hidden_layers", type=int, default=2, required=False,
                        help="Specifies maximum number of hidden layers in model.")
    parser.add_argument("-e", "--epochs", type=int, default=15, required=False,
                        help="Specifies number of epochs for training.")
    parser.add_argument("-m", "--num_of_models", type=int, default=4, required=False,
                        help="Specifies number of models to generate. If 1, returns an untrained model.")
    parser.add_argument("-i", "--interactive", action='store_true', required=False,
                        help="Specifies whether to enable interactive mode. If interactive mode is not enabled, all the models are trained.")
    args = vars(parser.parse_args())
    obj = RandomModelTrainer(num_hidden_layers=args['hidden_layers'],
                             num_epochs=args['epochs'],
                             num_models=args['num_of_models'])
    obj.generate_random_models()

    if not args['interactive']:
        # obj.view_models(models=None)
        obj.train_models(models=None)
    else:
        print("Instructions for usage:")
        print("'V indices' : View the models at indices less than number of models.")
        print("              indices -> None | integer[,integer]")
        print("'T indices' : Train the models at indices less than number of models.")
        print("              indices -> None | integer[,integer]")
        print("'S indices' : Save the models at indices less than number of models.")
        print("              indices -> None | integer[,integer]")
        print("'exit' : Exit the interactive session.")

        command = input("Enter instruction: ")
        while True:
            instr, indices = command.split(' ')
            if instr not in ["V", "T", "exit"]:
                print("Please enter proper input")
                command = input("Enter instruction: ")
                continue

            if len(indices) == 0:
                models = None
            elif len(indices) == 1:
                models = int(indices)
            else:
                models = [int(x) for x in indices.split(',')]
            
            if instr is "V":
                obj.view_models(models=models)
            elif instr is "T":
                obj.train_models(models=models)
            else:
                break

            command = input("Enter instruction: ")
