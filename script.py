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
        print("\n\nInstructions for usage:")
        print("'V [indices]' : View the model(s) specified by a list of indices")
        print("              indices -> None | integer[,integer]")
        
        print("'T [indices]' : Train the model(s) specified by a list of indices")
        print("              indices -> None | integer[,integer]")
        
        print("'VD indices' : View a random example from the datasets.")
        print("              indices -> train | test")

        print("'S [indices] path' : Save model(s) specified by a list of indices to the provided directory path")
        print("              indices -> None | integer[,integer]")
        print("              path -> string (absolute path)")
        
        print("'exit' : Exit the interactive session.")

        command = input("\n>>> Enter instruction: ")
        while True:
            if command == "exit":
                break

            try:
                instr, indices = command.split(' ')
            except ValueError:
                print("Please enter valid input")
                command = input("\n>>> Enter instruction: ")
                continue

            if instr not in ["V", "T", "S", "VD"]:
                print("Please enter valid input")
                command = input("\n>>> Enter instruction: ")
                continue

            if instr == "VD":
                dataset_choice = indices
                print("Choice is %s" % (dataset_choice))
            else:
                if len(indices) == 0:
                    models = None
                elif len(indices) == 1:
                    models = int(indices)
                else:
                    models = [int(x) for x in indices.split(',')]
            
            if instr == "V":
                obj.view_models(models=models)
            elif instr == "T":
                obj.train_models(models=models)
            elif instr == "VD":
                obj.view_dataset_example(dataset=dataset_choice)

            command = input("\n>>> Enter instruction: ")
